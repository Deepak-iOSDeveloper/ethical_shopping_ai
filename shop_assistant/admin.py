from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import SearchLog, FavoriteProduct, UserSavedProduct


@admin.register(UserSavedProduct)
class UserSavedProductAdmin(admin.ModelAdmin):
    list_display  = ('name', 'brand', 'category', 'eco_score', 'ethics_score',
                     'overall_score', 'carbon_level', 'user', 'source', 'saved_at')
    list_filter   = ('category', 'carbon_level', 'source')
    search_fields = ('name', 'brand', 'user__email', 'user__username')
    readonly_fields = ('saved_at',)
    ordering      = ('-saved_at',)


@admin.register(SearchLog)
class SearchLogAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'category', 'budget', 'results_count', 'timestamp')
    ordering     = ('-timestamp',)


@admin.register(FavoriteProduct)
class FavoriteProductAdmin(admin.ModelAdmin):
    list_display = ('product_name', 'brand', 'session_key', 'added_at')
